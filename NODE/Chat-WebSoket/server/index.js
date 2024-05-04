import express from 'express'
import logger from 'morgan'

//  importar socket.io
import { Server } from 'socket.io'
// importar http -> crear servidor http con express para socket.io
import { createServer } from 'node:http'

// definir puerto
const port = process.env.PORT ?? 3000

// iniciar express
const app = express()
// crear servidor http con express para socket.io
const server = createServer(app)
// crear servidor socket.io
const io = new Server(server, {
    // configuracion de socket.io para no perder la informacion de los usuarios al desconectarse
    connectionStateRecovery: {
        // 
    }
})

// cuando el io tenga conexion, ejecutar callback
io.on('connection', (socket) => {
    console.log('Usuario conectado')

    // cuando se desconecte el usuario, ejecutar callback
    socket.on('disconnect', () => {
        console.log('Usuario desconectado')
    })

    // cuando se reciba un mensaje, ejecutar callback
    socket.on('chat message', (message) => {
        // emitir mensaje a todos los usuarios
        io.emit('chat message', message)
    })
})

// usar morgan
app.use(logger('dev'))

//  iniciar ruta principal y servidor
app.get('/', (req, res) => {
    res.sendFile(process.cwd() + '/client/index.html')
})

// iniciar servidor
server.listen(port, () => {
    console.log(`Servidor en la ruta: http://localhost:${port}`)
})