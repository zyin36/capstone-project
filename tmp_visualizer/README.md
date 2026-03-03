download steps:

ensure you have node version v22.22.0. You can confirm this doing node -v in your terminal.

If not, you probably cannot run this visualizer.

The steps to get node version v22.22.0 is

1. Deleting the node_modules directory and package-lock.json file.
2. Executing curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
3. Executing source ~/.bashrc
4. Executing nvm install 22 && nvm use 22
5. Confirming via node -v
6. Executing npm install

Hopefully now, you can run the visualizer via npm run dev.