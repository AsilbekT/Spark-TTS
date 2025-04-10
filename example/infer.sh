#!/bin/bash

# Copyright (c) 2025 SparkAudio
#               2025 Xinsheng Wang (w.xinshawn@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Get the absolute path of the script's directory
script_dir=$(dirname "$(realpath "$0")")

# Get the root directory
root_dir=$(dirname "$script_dir")

# Set default parameters
device=1
save_dir='/home/asilbek/Desktop/work/assign/Spark-TTS/test_audio'
model_dir="pretrained_models/Spark-TTS-0.5B"
text="The issue you're describing, where the output audio file is taking too long to generate even if the input text is short, could be caused by several factors. Let's go through some potential causes and solutions based on the code you've provided."
gender="male"
pitch="moderate"
speed="moderate"
prompt_text="Hello my name is Asilbek, and this is test audio"
prompt_speech_path="/example/prompt.wav"

# Change directory to the root directory
cd "$root_dir" || exit

source sparktts/utils/parse_options.sh

# Run inference
python -m cli.inference \
    --text "${text}" \
    --gender "${gender}" \
    --device "${device}" \
    --save_dir "${save_dir}" \
    --model_dir "${model_dir}" \
    --pitch "${pitch}" \
    --speed "${speed}" \
    --prompt_speech_path "${prompt_speech_path}" \
    --prompt_text "${prompt_text}" 