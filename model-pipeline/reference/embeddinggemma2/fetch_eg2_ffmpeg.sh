#!/bin/sh
# Fetch the pinned FFmpeg shared builds used by the EmbeddingGemma 2 video spike and release gates.
#
# BtbN/FFmpeg-Builds autobuild of FFmpeg n8.1.3-17-gc389840894 (release branch 8.1, commit c389840894). The LGPL build
# supplies the decoder libraries that torchcodec (the upstream decoder) and the C# adapter load; that is all the gates
# need, and it is the default. BtbN deleted the autobuild, so the LGPL archive comes unchanged from the Starlinghq
# mirror, pinned by the mirror's commit (its README records the origin, licence and source).
#
# --with-gpl also requires the GPL build, whose encoders (libx264, libvpx-vp9) only generate the synthetic spike videos
# (make_eg2_videos.py). It is not mirrored. Put a copy in the destination directory: it is checked like the LGPL one.
# Each archive is checked by SHA-256.
#
# Usage: fetch_eg2_ffmpeg.sh [--with-gpl] <destination directory>
set -eu
variants=lgpl
if [ "${1:-}" = --with-gpl ]; then
  variants="gpl lgpl"
  shift
fi
dest="${1:?destination directory}"
mkdir -p "$dest"
cd "$dest"
mirror=https://huggingface.co/datasets/Starlinghq/ffmpeg-lgpl-shared-builds/resolve/ebbcfa02b5e6785acba4d8ad4361f320f94531af
rev=n8.1.3-17-gc389840894
for variant in $variants; do
  case "$variant" in
    gpl) want=4f58c09f117e63a286f2cc2da2d2934b1a45ddc5d00ba7d19151268a21cc16ba ;;
    lgpl) want=bf3db18a4a4bdc066f6b74ee5ec3d7b7be7d6738dcfd7162eae3e09cb857c6b2 ;;
  esac
  archive="ffmpeg-$rev-linux64-$variant-shared-8.1.tar.xz"
  if [ ! -f "$archive" ]; then
    if [ "$variant" = gpl ]; then
      echo "$archive is not in $dest. BtbN deleted this build and it is not mirrored; it is only needed to regenerate" >&2
      echo "the synthetic spike videos (make_eg2_videos.py). Copy it there, or run without --with-gpl for the LGPL build." >&2
      exit 1
    fi
    curl -fsSL --retry 3 -o "$archive" "$mirror/$archive"
  fi
  have=$(sha256sum "$archive" | cut -d' ' -f1)
  if [ "$have" != "$want" ]; then
    echo "SHA-256 mismatch for $archive: $have" >&2
    exit 1
  fi
  tar xf "$archive"
  echo "$archive $have"
done
