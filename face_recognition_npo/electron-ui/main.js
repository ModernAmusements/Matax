const { app, BrowserWindow, Menu, ipcMain } = require('electron');
const path = require('path');
const http = require('http');

let mainWindow;
const API_PORT = 3000;

function isPortInUse(port) {
    return new Promise((resolve) => {
        const net = require('net');
        const server = net.createServer();
        server.unref();
        server.on('error', () => {
            resolve(true);
        });
        server.on('listening', () => {
            server.close();
            resolve(false);
        });
        server.listen(port, '0.0.0.0');
    });
}

async function checkApiServer() {
    const portInUse = await isPortInUse(API_PORT);
    if (portInUse) {
        console.log(`Electron: Connected to existing API server on port ${API_PORT}`);
        return true;
    } else {
        console.error(`Electron Error: API server not found on port ${API_PORT}`);
        console.error('Please start the API server first: python api_server.py');
        return false;
    }
}

function createWindow() {
    mainWindow = new BrowserWindow({
        width: 1400,
        height: 900,
        minWidth: 1200,
        minHeight: 700,
        backgroundColor: '#F5F5F7',
        webPreferences: {
            nodeIntegration: false,
            contextIsolation: true,
            preload: path.join(__dirname, 'preload.js')
        },
        frame: true,
        titleBarStyle: 'hiddenInset',
        show: false
    });

    mainWindow.loadURL(`http://localhost:${API_PORT}`);

    mainWindow.once('ready-to-show', () => {
        mainWindow.show();
        mainWindow.focus();
    });

    mainWindow.on('closed', () => {
        mainWindow = null;
    });

    const menu = Menu.buildFromTemplate([
        {
            label: 'Face Recognition',
            submenu: [
                { label: 'About', click: () => showAbout() },
                { type: 'separator' },
                { role: 'services' },
                { type: 'separator' },
                { role: 'hide' },
                { role: 'hideOthers' },
                { role: 'unhide' },
                { type: 'separator' },
                { label: 'Quit', accelerator: 'CmdOrCtrl+Q', click: () => app.quit() }
            ]
        },
        {
            label: 'Edit',
            submenu: [
                { role: 'undo' },
                { role: 'redo' },
                { type: 'separator' },
                { role: 'cut' },
                { role: 'copy' },
                { role: 'paste' },
                { role: 'selectAll' }
            ]
        },
        {
            label: 'View',
            submenu: [
                { role: 'reload' },
                { role: 'forceReload' },
                { role: 'toggleDevTools' },
                { type: 'separator' },
                { role: 'resetZoom' },
                { role: 'zoomIn' },
                { role: 'zoomOut' },
                { type: 'separator' },
                { role: 'togglefullscreen' }
            ]
        },
        {
            label: 'Window',
            submenu: [
                { role: 'minimize' },
                { role: 'zoom' },
                { type: 'separator' },
                { role: 'front' }
            ]
        }
    ]);

    Menu.setApplicationMenu(menu);
}

function showAbout() {
    const { dialog } = require('electron');
    dialog.showMessageBox(mainWindow, {
        type: 'info',
        title: 'About Face Recognition',
        message: 'Face Recognition App',
        detail: 'NGO Facial Image Analysis System\n\nVersion 1.0.0\n\nAdvanced face detection and recognition powered by AI.'
    });
}

app.whenReady().then(async () => {
    const serverRunning = await checkApiServer();
    if (serverRunning) {
        createWindow();
    } else {
        // Still create window but it will show connection error
        createWindow();
    }
});

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
        createWindow();
    }
});
