# Dangin

Dangin is Danbots future inference and stitching engine.
Right now is on early stages of its development cycle, but the future looks bright :)

## Workflow
Im planning on implementing **Gitflow's workflow** to handle the development of features, bugs, etc. Please read more about it on the internet. KEEP always in mind that main branch is HOLY, a development branch and features branch should be the way and only correct way to work :)

Eventhough I wont be present at all in the following weeks, feel free to contact me on slack and I will take a look as soon as Im available. 

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

- **config.yaml**
    - Its a super simple but fast way to configure default behaviours. It is open for improvements.

- **src**
    - Is where dangine lives, inside resides the python module with all the components that find themselves on early stages.
    - **SUGGESTION:** I recommend you to explore the module to get familiarized of thecurrent implementation.

## Module's File Structure
The idea is to keep everything organize and well structure. An inspiration for dangin is pytorch structure.
At the moment the only submodules existing in dangin are:
- **loaders**
- **metrics** (currently empty)
- *core.py*
