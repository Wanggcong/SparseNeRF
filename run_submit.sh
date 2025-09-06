current_time=$(date +"%Y-%m-%d-%T")
echo "Current time: $current_time"
# git remote add homepage https://github.com/Wanggcong/Wanggcong.github.io.git(去网页github仓库上拷贝)，第一次需要特意去设置一下远端仓库的名字，和链接对应起来
# check local branch via the "git branch" comand, here is "master"： 在本地仓库（文件夹）下运行git branch
# check remote branch by check the repo of Github(去网页github仓库上看), here is "master"
git pull sparse main:main # git pull <remote> <remote_branch>:<local_branch>
git add .
git commit -m "$current_time"
git push -u sparse main:main # git push -u <remote> <local_branch>:<remote_branch>



