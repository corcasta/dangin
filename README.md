
# Dangin

Dangin is Danbots future inference and stitching engine.
Right now is on early stages of its development cycle, but the future looks bright :)


## Installation
During development just locate inside the root directory of the repo and run:
```bash
pip install -e .
```
Now any modifications done to dangin module will take effect on real time.

## Explanation
Dangine file structure is as follows:
- **datasets**
    - As the name implies is where all datasets should be allocated. The current datasets in the repo are: 
    **(All datasets except LAWAND are composed of 3 trajectories, divide the total number of files by 3 to read only 1 trajectory. Each trajectory represents the scanner moving through the teeth from a particular angle.)**

            1. LA100 
            2. LA200
            3. LA200_inf
            4. LA200_inf_v2
            4. LAWAND (REAL data comming from WAND)
- **notebooks**
  
    - Jupyter notebooks are great for testing new functionalities without the need deal with scripts calls. 
    - **SUGGESTION:** I highly recommend using notebook for development and testing. To load dangin after any modification just restart the notebook's kernel for them to take effect.

- **scripts** (doesn't exist yet)
    - Is the section intended for deploying finalized production level code related to stitching and in the future inference stuff.

- **src**
    - Is where dangine lives, inside resides the python module with all the components that find themselves on early stages.
    - **SUGGESTION:** I recommend you to explore the module to get familiarized of thecurrent implementation.


