document.addEventListener("DOMContentLoaded", async function() {
    const video = document.getElementById('video');
    const captureButton = document.getElementById('captureButton');

    navigator.mediaDevices.getUserMedia({ video: true })
        .then(function(stream) {
            video.srcObject = stream;
        })
        .catch(function(err) {
            console.error('Error accessing the camera: ', err);
        });

    captureButton.addEventListener('click', async function() {
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
        const imgBlob = await new Promise(resolve => canvas.toBlob(resolve, 'image/png'));

        try {
            const folderHandle = await showDirectoryPicker({ startIn: 'downloads' });
            const writable = await folderHandle.getFileHandle('captured-image.png', { create: true });
            const stream = await writable.createWritable();
            await stream.write(imgBlob);
            await stream.close();
            console.log('Image saved successfully.');

            // Assuming you have the python code in the same directory as your JavaScript file
            console.log('Spawning Python process');
            const pythonProcess = spawn('python', ['code2.py']);            
            pythonProcess.stdout.on('data', (data) => {
                console.log(`stdout: ${data}`);
            });
            pythonProcess.stderr.on('data', (data) => {
                console.error(`stderr: ${data}`);
            });
            pythonProcess.on('close', (code) => {
                console.log(`child process exited with code ${code}`);
            });
        } catch (err) {
            console.error('Error saving image:', err);
        }
    });
});
