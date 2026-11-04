#!/bin/sh
# Fetch the pinned FFmpeg shared builds used by the EmbeddingGemma 2 video spike and release gates.
#
# FFmpeg n8.1.3-17-gc389840894 (release branch 8.1, commit c389840894), LGPL. It supplies the decoder libraries that
# torchcodec (the upstream decoder) and the C# adapter load; that is all the gates need, and it is the default.
# build_eg2_ffmpeg.sh built it from pinned sources; the archive and all its source are in the Hugging Face dataset
# Starlinghq/ffmpeg-lgpl-rebuilds, pinned by the dataset's commit.
#
# --with-gpl also requires the BtbN/FFmpeg-Builds GPL build of the same FFmpeg commit, whose encoders (libx264,
# libvpx-vp9) only generate the synthetic spike videos (make_eg2_videos.py). BtbN deleted it and nothing here hosts it.
# Put a copy in the destination directory: it is checked like the LGPL one. Each archive is checked by SHA-256.
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
source=https://huggingface.co/datasets/Starlinghq/ffmpeg-lgpl-rebuilds/resolve/PIN_HF
rev=n8.1.3-17-gc389840894
for variant in $variants; do
  case "$variant" in
    gpl)
      want=4f58c09f117e63a286f2cc2da2d2934b1a45ddc5d00ba7d19151268a21cc16ba
      archive="ffmpeg-$rev-linux64-gpl-shared-8.1.tar.xz"
      ;;
    lgpl)
      want=PIN_SHA8
      archive="ffmpeg-$rev-starling-linux64-lgpl-shared-8.1.tar.xz"
      ;;
  esac
  if [ ! -f "$archive" ]; then
    if [ "$variant" = gpl ]; then
      echo "$archive is not in $dest. BtbN deleted this build and nothing here hosts it; it is only needed to regenerate" >&2
      echo "the synthetic spike videos (make_eg2_videos.py). Copy it there, or run without --with-gpl for the LGPL build." >&2
      exit 1
    fi
    curl -fsSL --retry 3 -o "$archive" "$source/$archive"
  fi
  have=$(sha256sum "$archive" | cut -d' ' -f1)
  if [ "$have" != "$want" ]; then
    echo "SHA-256 mismatch for $archive: $have" >&2
    exit 1
  fi
  tar xf "$archive"
  echo "$archive $have"
done
