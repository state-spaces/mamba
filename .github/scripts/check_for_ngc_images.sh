#!/bin/bash

# Configuration
BASE_IMAGE="nvcr.io/nvidia/pytorch"
TAG_SUFFIX="-py3"
MONTHS_TO_CHECK=7 # Check current month and previous 6 months (total 7)

# Initialize an array to store existing tags
EXISTING_TAGS=()

echo "Checking for existence of the last ${MONTHS_TO_CHECK} NGC PyTorch images: ${BASE_IMAGE}:YY.MM${TAG_SUFFIX}"
echo "---------------------------------------------------------------------"

# Loop through the last N months
for i in $(seq 0 $((MONTHS_TO_CHECK - 1))); do
    # Calculate Year and Month for the tag
    CURRENT_YEAR=$(date +%Y)
    CURRENT_MONTH=$(date +%m)
    
    # Calculate target month and year
    TARGET_DATE=$(date -d "$CURRENT_YEAR-$CURRENT_MONTH-01 -$i months" +%y.%m)
    
    # Construct the full image tag and the tag-only string
    IMAGE_TAG="${TARGET_DATE}${TAG_SUFFIX}"
    FULL_IMAGE="${BASE_IMAGE}:${IMAGE_TAG}"

    echo "Checking: ${FULL_IMAGE}"

    # Use 'docker manifest inspect' to check for image existence without pulling.
    if docker manifest inspect "${FULL_IMAGE}" > /dev/null 2>&1; then
        echo "✅ EXISTS: Found."
        # Add the tag-only string to the array
        EXISTING_TAGS+=("nvcr.io/nvidia/pytorch:${IMAGE_TAG}")
    else
        echo "❌ MISSING: Not found."
    fi
done

echo "---------------------------------------------------------------------"

## JSON Output Generation
# This uses the collected array to build a JSON string.

# 1. Convert the shell array to a newline-separated string.
TAGS_NL_SEP=$(printf "%s\n" "${EXISTING_TAGS[@]}")

# 2. Use jq to read the newline-separated list and format it into a JSON array.
# . | split("\n") | .[:-1] reads the input, splits it by newline, and removes the trailing empty element.
if command -v jq &> /dev/null; then
    JSON_STRING=$(echo -e "${TAGS_NL_SEP}" | jq -R -s 'split("\n") | .[:-1]')
    
    echo "Generated JSON String of Existing Tags:"
    echo "${JSON_STRING}"
    
    # Optional: Save the JSON string to a variable for further use
    # echo "JSON_STRING is now available in the shell if you source this script."
else
    echo "WARNING: 'jq' is not installed. Cannot format output as JSON."
    echo "Found Tags: ${EXISTING_TAGS[*]}"
fi

echo "---"
echo "Check complete."

echo "${JSON_STRING}" > ngc_images.json