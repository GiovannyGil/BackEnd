import express from "express";


const app = express() // inicializaar express


// crear ruta para la pagina inicial
app.get('/', (req, res) => {
    res.send('hello world')
})

app.listen(3000) // iniciar el servidor local
console.log('Servidor corriendo en el puerto: ', 3000)

