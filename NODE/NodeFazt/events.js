const EventEmitter = require('events') 


const customEmitter = new EventEmitter() // declarar un nuevo evento

customEmitter.on('response', (data)=>{ // "escuchar" el evento response y recibir data
    console.log('recibido')
    console.log(data)
})
// emitir un evento
customEmitter.emit('response', 'Hola Mundo') // parametro 'response' y parametro 'data'