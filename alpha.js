const express = require('express');
const bodyParser = require('body-parser');
const fs = require('fs');
const path = require('path');
const { v4: uuidv4 } = require('uuid');

const app = express();
const PORT = 3000;

app.use(bodyParser.json());

app.post('/save-image', (req, res) => {
    const imageDataURL = req.body.image;
    const base64Data = imageDataURL.replace(/^data:image\/png;base64,/, '');
    const filename = `${uuidv4()}.png`;
    const filePath = path.join(__dirname, 'images', filename);

    fs.writeFile(filePath, base64Data, 'base64', (err) => {
        if (err) {
            console.error('Error saving image:', err);
            res.status(500).json({ error: 'Failed to save image' });
        } else {
            console.log('Image saved:', filename);
            res.json({ filename });
        }
    });
});

app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});
