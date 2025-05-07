// server.js
const express = require('express');
const http = require('http');
const { Server } = require('socket.io');

const app = express();
const server = http.createServer(app);
const io = new Server(server);

// Serve your front-end
app.use(express.static('public'));

io.on('connection', socket => {
  console.log('Client connected:', socket.id);

  socket.on('frame', data => {
    // data is a Buffer of JPEG bytes
    socket.broadcast.emit('frame', data);       // rebroadcast to all others :contentReference[oaicite:11]{index=11}
  });

  socket.on('disconnect', () => {
    console.log('Client disconnected:', socket.id);
  });
});

server.listen(3000, () => console.log('Listening on :3000'));
