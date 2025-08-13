// main.js
const { app, BrowserWindow, ipcMain } = require('electron');
const { spawn } = require('child_process');
const path = require('path');

function resolvePythonCmd() {
  // Allow override: set PYTHON env var to a full path if you like
  if (process.env.PYTHON) return process.env.PYTHON;
  // On Windows, "python" or "py -3"; on macOS/Linux, "python3" is common
  return process.platform === 'win32' ? 'python' : 'python3';
}

function createWindow() {
  const win = new BrowserWindow({
    width: 900,
    height: 700,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  win.loadFile('index.html');
}

app.whenReady().then(() => {
  createWindow();
  app.on('activate', () => { if (BrowserWindow.getAllWindows().length === 0) createWindow(); });
});

app.on('window-all-closed', () => { if (process.platform !== 'darwin') app.quit(); });

ipcMain.handle('predict-image', async (_event, arrayBuffer) => {
  return new Promise((resolve, reject) => {
    const pythonCmd = resolvePythonCmd();
    const scriptPath = path.join(__dirname, 'predict_stdin.py');

    const py = spawn(pythonCmd, [scriptPath], { cwd: __dirname });

    let stdout = '';
    let stderr = '';

    py.stdout.on('data', d => { stdout += d.toString(); });
    py.stderr.on('data', d => { stderr += d.toString(); });

    py.on('error', (err) => {
      // e.g., Python not found
      reject(new Error(`Failed to start Python (${pythonCmd}): ${err.message}`));
    });

    py.on('close', code => {
      if (code !== 0) {
        return reject(new Error(stderr || `Python exited with ${code}`));
      }
      resolve(stdout.trim()); // e.g., "cat,0.9823"
    });

    // Write image bytes to Python stdin
    try {
      const buf = Buffer.from(arrayBuffer);
      py.stdin.write(buf);
      py.stdin.end();
    } catch (e) {
      reject(new Error(`Failed to send image bytes to Python: ${e.message}`));
    }
  });
});
