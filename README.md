# UCLA CS269 Spring2022 human AI Course Survey

Project page: https://ucladeepvision.github.io/CS269-surveys-2022spring/

## Instruction for running this site locally

1. Follow the first 2 steps in [pull-request-instruction](pull-request-instruction.md)

2. Installing Ruby with version 3.0.0 if you are using a Mac, and ruby 2.7 should work for Linux, check https://www.ruby-lang.org/en/documentation/installation/ for instruction.

3. Installing Bundler and jekyll with
```
gem install --user-install bundler jekyll
bundler install
bundle add webrick
```

4. Run your site with
```
bundle exec jekyll serve
```
You should see an address pop on the terminal (http://127.0.0.1:4000/CS269-surveys-2022spring
/ by default), go to this address with your browser.

## Working on the project

1. Create a folder with your team id under ```./assets/images/your-module-id```, for example, ```./assets/images/module01```. You will use this folder to store all the images in your project.

2. We have already created the .md file for you and you can work on them directly. Change the authors at the top to your team members. You may notice the year is 2021 in date attribute, that's intentionally and please do not correct it. We will fix it at the end of the semester.

3. Look at the .md file we provided and it contains basic elements that you might use in Markdown. We also provide a sample post and you can check the source code under ```./_posts```.

4. Start your work in your .md file. You may only edit the .md file you just and add images to ```./assets/images/your-module-id```. Please do **NOT** change any other files in this repo.

Once you save the .md file, jekyll will synchronize the site and you can check the changes on browser.

## Submission
We will use git pull request to manage submissions.

Once you've done, follow steps 3 and 4 in [pull-request-instruction](pull-request-instruction.md) to make a pull request BEFORE the deadline. Please make sure not to modify any file except your .md file and your images folder. We will merge the request after all submissions are received, and you should able to check your work in the project page on next week of each deadline.
