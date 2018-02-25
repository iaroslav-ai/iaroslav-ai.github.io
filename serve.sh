sudo apt-get install ruby-dev -y
sudo gem install jekyll bundler
printf "source 'https://rubygems.org' \ngem 'github-pages', group: :jekyll_plugins" > Gemfile
bundle install
bundle exec jekyll serve
