## Every Morning

Check out our Workflow at Miro ([link](https://miro.com/welcomeonboard/WVNmaGNCMzlnK1FjbDlYc2tKK1F4T3Y3K0s4Tm13MnBSY2JibElneXFzK2tBM0FPWStINXRzSHdNY1JqMCs5a3ZvQy90NVZBV3dBT3JPMTFyTTBYcjFCQ1ZPVFl1dm5ScHRJUWFEZ01pS0t2eE4wdWJObzczUTJCSVRGbSttQkJhWWluRVAxeXRuUUgwWDl3Mk1qRGVRPT0hdjE=?share_link_id=462218411826))

Check our Trello board([link](https://trello.com/invite/b/6936dbc891a1f358b51364db/ATTIf0c9ea7c184b84ac2f053b107edc3b201355B91E/my-trello-board)) for the tasks, check your progress

`cd ~/code/sasha17demin/hypopredict` (navigate into the project folder, check the pyenv == 'hyperpredict')

From `master`

`git pull origin master` (make sure you have updated repo)

`git checkout -b name_task` (moving to your working branch before making edits)


`pip install --upgrade pip`

`pip install -r requirements.txt` [installing new packages that others found useful will save you time]


## CheatSheet

`git branch` -- show all your branches

`git branch -D branch_name` -- deleting a branch





## Day 2: Generate Some Features + Train Our First Models

### Part 1:

* Move Notebooks to `notebooks` folder
* Use the code from `d1_get_familiar` to load the data and play with it
* Decide what model architecture (target, features) would you wanna do?

* Generate Chunk 1: 2 ppl all days all features from D1NAMO
* Write a train_test_split.py


### Part 2:

* train_test_split the Chunk 1
* everyone preprocesses Chunk 1
* builds baseline version of Modeil_i (Chunk 1)




## Day 1: Playing with the Dataset


'd1_get_familiar' folder contains data on Type-1 Diabetic (T1D) person id 01 day 2014-10-04:

1/ "data/get_familiar" folder with `.feather` files for different features related to hypoglycemia

2/ simple loading Notebook -- use it to explore

Suggested exploration tasks:

1. Load the data and make sure it reads well
2. Plot some of the things, understand it, try top merge it, pay attention to the time index
3. Create some descriptive statistics and understand what they mean for time-series context: rolling Heart Rate for example, what does that mean in code?
4. Read [this paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0325956) for an example architecture
