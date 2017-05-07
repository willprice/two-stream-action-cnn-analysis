#!/usr/bin/env bash
set -e

DATA_SOURCE="${1:-ucf101/test-1}"
EXAMPLE_LIST="${2:-example_list_ucf101}"

sed 1d "${EXAMPLE_LIST}" | while IFS=, read video spatial temporal start stop step
do
    echo "$video $spatial $temporal $start $stop $step"
    if [[ "$temporal" = t && "$spatial" != t ]]; then
        ./image_join.py --start "$start" \
                        --stop "$stop" \
                        --step "$step" \
                        --no-spatial \
                        "${DATA_SOURCE}" \
                        "$video"
        mv "${video}.pdf" "${video}-temporal.pdf"
    elif [[ "$spatial" = t && "$temporal" != t ]]; then
        ./image_join.py --start "$start" \
                        --stop "$stop" \
                        --step "$step" \
                        --no-temporal \
                        "${DATA_SOURCE}" \
                        "$video"
        mv "${video}.pdf" "${video}-spatial.pdf"

    fi
    ./image_join.py --start "$start" \
                    --stop "$stop" \
                    --step "$step" \
                    "${DATA_SOURCE}" \
                    "$video"
done
