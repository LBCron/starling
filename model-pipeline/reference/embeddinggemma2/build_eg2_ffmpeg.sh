#!/usr/bin/env bash
# Build the LGPL shared FFmpeg libraries of the EmbeddingGemma 2 video gates from pinned sources.
#
# FFmpeg at the two commits the adapters' bindings and the protected references name, with its own decoders, x86 assembly
# on, and one external library: dav1d, because FFmpeg has no software AV1 decoder of its own and the fixtures and the
# retrieval set hold AV1 videos. dav1d is linked statically into libavcodec, so lib/ holds FFmpeg's libraries only.
# --disable-autodetect keeps everything else out, whatever the build machine has installed. The result is
# "LGPL version 2.1 or later": no --enable-gpl, and no --enable-version3 because dav1d (BSD-2-Clause) does not need it.
#
#   build_eg2_ffmpeg.sh sources <source dir>                  source archives from upstream, needs git and network
#   build_eg2_ffmpeg.sh build <source dir> <out dir> [jobs]   both builds in a container without network, needs Docker
#
# `build` takes the archives `sources` wrote or the same files from Starlinghq/ffmpeg-lgpl-rebuilds, checks them by
# SHA-256 and writes the two archives with their digests and logs. fetch_eg2_ffmpeg.sh and the gates use those archives.
set -euo pipefail

ffmpeg8_commit=c3898408946f6cbfe8e244372c250366d5639d00
ffmpeg8_describe=n8.1.3-17-gc389840894
ffmpeg8_series=8.1
ffmpeg8_source_sha256=8d40cfdd532853f299cd75d20b3ac04904921cc9aa652b2ec6684444886497fc
ffmpeg9_commit=669f5ab083c7732c10f1ce9e4f1915960dd88457
ffmpeg9_describe=n9.0.2-28-g669f5ab083
ffmpeg9_series=9.0
ffmpeg9_source_sha256=974b3940f00b5a5ab85ebc3c7bd3260be9a6c542db41cd18b83f1fd3040afb4c
dav1d_commit=54706fc6bc0cdecab7e9593974a4039cc038fca7 # tag 1.5.4
dav1d_source_sha256=fd305559224bee4fa2d5c11a36fb88957bfec754f3b567d8f0548b96bb3e5c67
ffmpeg_url=https://github.com/FFmpeg/FFmpeg.git
dav1d_url=https://code.videolan.org/videolan/dav1d.git

# Every option but the prefix, which is /opt/ffmpeg. The rpath lets bin/ffmpeg find ../lib.
configure_options=(
    --arch=x86_64
    --target-os=linux
    --disable-autodetect
    --disable-iconv
    --enable-shared
    --disable-static
    --disable-debug
    --disable-doc
    --enable-libdav1d
    --pkg-config-flags=--static
    '--extra-ldexeflags=-Wl,--disable-new-dtags,-rpath,\$$ORIGIN/../lib'
)

check_sha256() { # <file> <expected>
    local have
    have=$(sha256sum "$1" | cut -d' ' -f1)
    [[ "$have" == "$2" ]] || { echo "SHA-256 mismatch for $1: $have, expected $2" >&2; exit 1; }
}

# The archive is `git archive` of the commit. A build from it reads its version from the VERSION file, as release
# tarballs do, so this is where the git describe that file will carry is checked.
sources_ffmpeg() { # <clone> <out> <commit> <describe>
    local have
    have=$(git -C "$1" describe --tags --always "$3")
    [[ "$have" == "$4" ]] || { echo "git describe of $3 is $have, expected $4" >&2; exit 1; }
    git -C "$1" archive --format=tar --prefix="ffmpeg-$4/" "$3" | xz -9 -T1 >"$2/ffmpeg-$3.tar.xz"
}

cmd_sources() {
    local out=$1
    mkdir -p "$out"
    tmp=$(mktemp -d)
    trap 'rm -rf "$tmp"' EXIT
    git clone --quiet --bare --filter=blob:none "$ffmpeg_url" "$tmp/ffmpeg.git"
    sources_ffmpeg "$tmp/ffmpeg.git" "$out" "$ffmpeg8_commit" "$ffmpeg8_describe"
    sources_ffmpeg "$tmp/ffmpeg.git" "$out" "$ffmpeg9_commit" "$ffmpeg9_describe"
    git clone --quiet --bare --filter=blob:none "$dav1d_url" "$tmp/dav1d.git"
    git -C "$tmp/dav1d.git" archive --format=tar --prefix="dav1d-$dav1d_commit/" "$dav1d_commit" |
        xz -9 -T1 >"$out/dav1d-$dav1d_commit.tar.xz"
    (cd "$out" && sha256sum ffmpeg-*.tar.xz dav1d-*.tar.xz)
}

cmd_build() {
    local sources out jobs here tag name
    sources=$(cd "$1" && pwd)
    mkdir -p "$2"
    out=$(cd "$2" && pwd)
    jobs=${3:-6}
    here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
    check_sha256 "$sources/ffmpeg-$ffmpeg8_commit.tar.xz" "$ffmpeg8_source_sha256"
    check_sha256 "$sources/ffmpeg-$ffmpeg9_commit.tar.xz" "$ffmpeg9_source_sha256"
    check_sha256 "$sources/dav1d-$dav1d_commit.tar.xz" "$dav1d_source_sha256"
    tag=starling-ffmpeg-lgpl-build:$(sha256sum "$here/build_eg2_ffmpeg.Dockerfile" | cut -c1-12)
    docker build --quiet --tag "$tag" - <"$here/build_eg2_ffmpeg.Dockerfile" >/dev/null
    name=starling-ffmpeg-lgpl-build-$$
    docker run --rm --name "$name" --network=none --memory=20g --memory-swap=20g --cpus="$jobs" \
        --user "$(id -u):$(id -g)" --tmpfs /work:exec,size=12g --env "JOBS=$jobs" \
        --volume "$sources:/sources:ro" --volume "$out:/out" \
        --volume "$here/build_eg2_ffmpeg.sh:/build_eg2_ffmpeg.sh:ro" \
        "$tag" bash /build_eg2_ffmpeg.sh inside
    (cd "$out" && sha256sum ffmpeg-*.tar.xz | tee SHA256SUMS)
}

# In the container.
inside_ffmpeg() { # <commit> <describe> <series>
    local commit=$1 describe=$2 series=$3 name lib prefix=/opt/ffmpeg
    name=ffmpeg-$describe-starling-linux64-lgpl-shared-$series
    tar xf "/sources/ffmpeg-$commit.tar.xz" -C /work
    (
        cd "/work/ffmpeg-$describe"
        echo "$describe" >VERSION
        PKG_CONFIG_PATH=/work/dav1d-prefix/lib/pkgconfig ./configure --prefix="$prefix" "${configure_options[@]}" \
            | tee "/out/$name.configure.log"
        grep -q '^#define HAVE_X86ASM 1$' config.h
        make -j"$JOBS" >"/out/$name.make.log" 2>&1
        make install DESTDIR="/work/$name.root" >>"/out/$name.make.log" 2>&1
    )
    # External libraries as configure reports them: dav1d alone, and nothing for hardware acceleration.
    if [[ $(sed -n '/^External libraries:$/,/^$/p' "/out/$name.configure.log" | sed '1d;/^$/d' | xargs) != libdav1d ]] ||
        [[ -n $(sed -n '/^External libraries providing hardware acceleration:$/,/^$/p' "/out/$name.configure.log" | sed '1d;/^$/d') ]]; then
        echo "configure enabled other external libraries, see $name.configure.log" >&2
        exit 1
    fi
    mkdir -p "/work/$name/LICENSES"
    cp -a "/work/$name.root$prefix/." "/work/$name/"
    cp "/work/ffmpeg-$describe/LICENSE.md" "/work/ffmpeg-$describe/COPYING.LGPLv2.1" "/work/$name/LICENSES/"
    cp "/work/dav1d-$dav1d_commit/COPYING" "/work/$name/LICENSES/COPYING.dav1d"
    {
        echo "FFmpeg $describe, commit $commit"
        echo "dav1d 1.5.4, commit $dav1d_commit, linked statically into libavcodec"
        echo "Source: https://huggingface.co/datasets/Starlinghq/ffmpeg-lgpl-rebuilds"
        echo "Built by build_eg2_ffmpeg.sh on Ubuntu 22.04 with:"
        printf '%q ' ./configure --prefix="$prefix" "${configure_options[@]}"
        echo
        echo "Toolchain packages:"
        cat /toolchain-packages.txt
    } >"/work/$name/BUILD-INFO.txt"
    # Same bytes on every build: fixed order, times and owners.
    tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner -C /work -cf - "$name" | xz -9 -T1 >"/out/$name.tar.xz"
    {
        for lib in "/work/$name"/lib/*.so.*.*.*; do
            echo "== ${lib##*/}"
            readelf -d "$lib" | grep -E 'NEEDED|RPATH|RUNPATH|SONAME'
        done
        echo "== newest glibc symbol version needed"
        readelf --dyn-syms --wide "/work/$name"/lib/*.so.*.*.* "/work/$name"/bin/ffmpeg |
            grep -oE 'GLIBC_[0-9.]+' | sort -Vu | tail -1
    } | tee "/out/$name.elf.txt"
    if grep NEEDED "/out/$name.elf.txt" | grep -vE '\[(libav[a-z]+|libsw[a-z]+|libm|libc|libdl|libpthread|librt)\.so\.[0-9]+\]$'; then
        echo "a library needs something outside FFmpeg and glibc, see above" >&2
        exit 1
    fi
    LD_LIBRARY_PATH=/work/$name/lib "/work/$name/bin/ffmpeg" -hide_banner -buildconf >"/out/$name.buildconf.txt" 2>&1
    rm -rf "/work/ffmpeg-$describe" "/work/$name" "/work/$name.root"
}

cmd_inside() {
    tar xf "/sources/dav1d-$dav1d_commit.tar.xz" -C /work
    meson setup /work/dav1d-build "/work/dav1d-$dav1d_commit" --prefix=/work/dav1d-prefix --libdir=lib \
        --buildtype=release -Ddefault_library=static -Denable_tools=false -Denable_tests=false >/out/dav1d.log 2>&1
    ninja -C /work/dav1d-build -j"$JOBS" install >>/out/dav1d.log 2>&1
    inside_ffmpeg "$ffmpeg8_commit" "$ffmpeg8_describe" "$ffmpeg8_series"
    inside_ffmpeg "$ffmpeg9_commit" "$ffmpeg9_describe" "$ffmpeg9_series"
}

case "${1:-}" in
    sources) shift; cmd_sources "$@" ;;
    build) shift; cmd_build "$@" ;;
    inside) cmd_inside ;;
    *) sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d' >&2; exit 2 ;;
esac
