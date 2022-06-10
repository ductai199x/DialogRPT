# step 0. create the data folder

mkdir "data/compressed"

# Step 1. Download raw data from a third party dump: https://files.pushshift.io/reddit

# download comments for year 2011
wget https://files.pushshift.io/reddit/comments/RC_2011-01.bz2 -P data/compressed
wget https://files.pushshift.io/reddit/comments/RC_2011-02.bz2 -P data/compressed
wget https://files.pushshift.io/reddit/comments/RC_2011-03.bz2 -P data/compressed
wget https://files.pushshift.io/reddit/comments/RC_2011-04.bz2 -P data/compressed
wget https://files.pushshift.io/reddit/comments/RC_2011-05.bz2 -P data/compressed
wget https://files.pushshift.io/reddit/comments/RC_2011-06.bz2 -P data/compressed
wget https://files.pushshift.io/reddit/comments/RC_2011-07.bz2 -P data/compressed
wget https://files.pushshift.io/reddit/comments/RC_2011-08.bz2 -P data/compressed
wget https://files.pushshift.io/reddit/comments/RC_2011-09.bz2 -P data/compressed
wget https://files.pushshift.io/reddit/comments/RC_2011-10.bz2 -P data/compressed
wget https://files.pushshift.io/reddit/comments/RC_2011-11.bz2 -P data/compressed
wget https://files.pushshift.io/reddit/comments/RC_2011-12.bz2 -P data/compressed

# download comments for year 2012
wget https://files.pushshift.io/reddit/comments/RC_2012-01.bz2 -P data/compressed
wget https://files.pushshift.io/reddit/comments/RC_2012-02.bz2 -P data/compressed
wget https://files.pushshift.io/reddit/comments/RC_2012-03.bz2 -P data/compressed
wget https://files.pushshift.io/reddit/comments/RC_2012-04.bz2 -P data/compressed
wget https://files.pushshift.io/reddit/comments/RC_2012-05.bz2 -P data/compressed
wget https://files.pushshift.io/reddit/comments/RC_2012-06.bz2 -P data/compressed
wget https://files.pushshift.io/reddit/comments/RC_2012-07.bz2 -P data/compressed
wget https://files.pushshift.io/reddit/comments/RC_2012-08.bz2 -P data/compressed
wget https://files.pushshift.io/reddit/comments/RC_2012-09.bz2 -P data/compressed
wget https://files.pushshift.io/reddit/comments/RC_2012-10.bz2 -P data/compressed
wget https://files.pushshift.io/reddit/comments/RC_2012-11.bz2 -P data/compressed
wget https://files.pushshift.io/reddit/comments/RC_2012-12.bz2 -P data/compressed

# download submissions for year 2011
wget https://files.pushshift.io/reddit/submissions/RS_2011-01.zst -P data/compressed
wget https://files.pushshift.io/reddit/submissions/RS_2011-02.zst -P data/compressed
wget https://files.pushshift.io/reddit/submissions/RS_2011-03.zst -P data/compressed
wget https://files.pushshift.io/reddit/submissions/RS_2011-04.zst -P data/compressed
wget https://files.pushshift.io/reddit/submissions/RS_2011-05.zst -P data/compressed
wget https://files.pushshift.io/reddit/submissions/RS_2011-06.zst -P data/compressed
wget https://files.pushshift.io/reddit/submissions/RS_2011-07.zst -P data/compressed
wget https://files.pushshift.io/reddit/submissions/RS_2011-08.zst -P data/compressed
wget https://files.pushshift.io/reddit/submissions/RS_2011-09.zst -P data/compressed
wget https://files.pushshift.io/reddit/submissions/RS_2011-10.zst -P data/compressed
wget https://files.pushshift.io/reddit/submissions/RS_2011-11.zst -P data/compressed
wget https://files.pushshift.io/reddit/submissions/RS_2011-12.zst -P data/compressed

# download submissions for year 2011
wget https://files.pushshift.io/reddit/submissions/RS_2012-01.zst -P data/compressed
wget https://files.pushshift.io/reddit/submissions/RS_2012-02.zst -P data/compressed
wget https://files.pushshift.io/reddit/submissions/RS_2012-03.zst -P data/compressed
wget https://files.pushshift.io/reddit/submissions/RS_2012-04.zst -P data/compressed
wget https://files.pushshift.io/reddit/submissions/RS_2012-05.zst -P data/compressed
wget https://files.pushshift.io/reddit/submissions/RS_2012-06.zst -P data/compressed
wget https://files.pushshift.io/reddit/submissions/RS_2012-07.zst -P data/compressed
wget https://files.pushshift.io/reddit/submissions/RS_2012-08.zst -P data/compressed
wget https://files.pushshift.io/reddit/submissions/RS_2012-09.zst -P data/compressed
wget https://files.pushshift.io/reddit/submissions/RS_2012-10.zst -P data/compressed
wget https://files.pushshift.io/reddit/submissions/RS_2012-11.zst -P data/compressed
wget https://files.pushshift.io/reddit/submissions/RS_2012-12.zst -P data/compressed

# Step 2. Read the compressed files and group items from the same subreddit 

python src/data.py -y 2011 -j
python src/data.py -y 2012 -j

# Step 3. extract basic attributes and dialog trees.

python src/data.py -y 2011 -b
python src/data.py -y 2011 -b

# Step 4. Build training and testing data for different feedback signals. 

python src/data.py -y 2011 -p
python src/data.py -y 2012 -p
