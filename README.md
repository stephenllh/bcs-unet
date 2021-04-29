# BCS-UNet

<!-- ABOUT THE PROJECT -->
## About The Project

<br/>
<p align="center">
  <img src="/image/image.png" alt="Competition image" width="500" height="500"/>
</p>


<!-- ![Product Name Screen Shot](/image/image.png) -->

This is my solution to the [Bengali.AI Handwritten Grapheme Classification competition](https://www.kaggle.com/c/bengaliai-cv19/overview). I used a DenseNet-121 that receives a one-channel image and outputs 3 separate classifiers for the grapheme root, vowel diacritics, and consonant diacritics, respectively. The MixUp augmentation is used to improve the performance.

Result: Macro-average recall score of 0.9319 in the [private leaderboard](https://www.kaggle.com/c/bengaliai-cv19/leaderboard). Ranked 182 out of 2059 teams (bronze medal region).
<br/><br/>

<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running, follow these simple example steps.
<br/><br/>

### Prerequisites

* PyTorch (version 1.3.0)

  Install using Anaconda:
  ```sh
  conda install pytorch=1.3.0 -c pytorch
  ```

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/stephenllh/bengali_character.git
   ```

1. Change directory
   ```sh
   cd bengali_character
   ```

2. Install packages
   ```sh
   pip install requirements.txt
   ```
<br/>

<!-- USAGE EXAMPLES -->
## Usage

1. Change directory
   ```sh
   cd bengali_character
   ```

2. Create a directory called `input`
   ```sh
   mkdir input
   cd input
   ```

3. Download the dataset into the folder
    - Option 1: Use Kaggle API
      - `pip install kaggle`
      - `kaggle competitions download -c bengaliai-cv19`
    - Option 2: Download the dataset from the [competition website](https://www.kaggle.com/c/bengaliai-cv19/data).

4. Run the training script
   ```sh
   cd ..
   python train.py
   ```

5. (Optional) Run the inference script
   ```sh
   python inference.py
   ```

<br/>


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.
<br></br>


<!-- CONTACT -->
## Contact

Stephen Lau - [Email](stephenlaulh@gmail.com) - [Twitter](https://twitter.com/StephenLLH) - [Kaggle](https://www.kaggle.com/faraksuli)


