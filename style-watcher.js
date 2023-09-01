const fs = require('fs');
const chokidar = require('chokidar');
const path = require('path');

// Initialize watcher.
const watcher = chokidar.watch(path.resolve(__dirname, './js/output.css'), {
  persistent: true
});

// Add event listeners.
watcher
  .on('change', filePath => {
    console.log(`File ${filePath} has been changed`);
    let cssContent = fs.readFileSync(filePath, 'utf8');
    let jsContent = `let style = document.createElement('style');
style.innerHTML = \`${cssContent.replace(/\\/g, '\\\\')}\`;
document.head.appendChild(style);`;
    fs.writeFileSync(path.resolve(__dirname, './js/style-inject.js'), jsContent);
  });

// On script exit, close the watcher
process.on('exit', (code) => {
  watcher.close();
});
