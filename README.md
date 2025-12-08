## Every Morning

Check our Trello board([link](https://trello.com/invite/b/6936dbc891a1f358b51364db/ATTIf0c9ea7c184b84ac2f053b107edc3b201355B91E/my-trello-board)) for the tasks, check your progress

`cd ~/code/sasha17demin/hypopredict` (navigate into the project folder, check the pyenv == 'hyperpredict')

`pip install --upgrade pip`

`pip install -r requirements.txt` [installing new packages that others found useful will save you time]

From `master`

`git pull origin master` (make sure you have updated repo)

`git checkout -b name_task` (moving to your working branch before making edits)




## Day 2: Generate Some Features + Train Our First Models



## Day 1: Playing with the Dataset


'd1_get_familiar' folder contains data on Type-1 Diabetic (T1D) person id 01 day 2014-10-04:

1/ "data/get_familiar" folder with `.feather` files for different features related to hypoglycemia

2/ simple loading Notebook -- use it to explore

Suggested exploration tasks:

1. Load the data and make sure it reads well
2. Plot some of the things, understand it, try top merge it, pay attention to the time index
3. Create some descriptive statistics and understand what they mean for time-series context: rolling Heart Rate for example, what does that mean in code?
4. Read [this paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0325956) for an example architecture
