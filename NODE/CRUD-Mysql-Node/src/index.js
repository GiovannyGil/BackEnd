import express from 'express'
import path from 'path'




const app = express() // iniciarlizar express
const port = 3000 // elegir puerto


// SETTINGS
app.set('port', process.env.PORT || port) // va a tomar el sistema que le asigne el S.O si no exite, tomaras el pueto : "port:3000"

// views
app.set('view engine', 'ejs') // elegir el motor de plantillas
app.set('views', path.join(__dirname, 'views'))


app.get('/', (req, res) => res.send('Hello World!')) // dar un funcion de entrada a la página principal
app.listen(app.get('port'), () => console.log(`Servidor Corriendo en el Puerto ${port}!`)) // correr el servidor