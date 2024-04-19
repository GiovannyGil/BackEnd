import  Express  from "express" // improtar express para crear un servidor
// import ejs from 'ejs'
import {dirname, join} from 'path'
import {fileURLToPath} from 'url'


import router from "./routes/index.js"

const app = Express() // inicializar express

const __dirname = dirname(fileURLToPath(import.meta.url)) // establecer ruta absoluta de la carpeta de las vistas

app.set('views', join(__dirname, 'views')) // ubicacion/ ruta de las vistas
app.set('motos de plantillas', 'ejs') // establecer con que motor de plantillas html se va a trabajar
app.use(router) // usar enrutador


app.listen(3000) // puerto donde va a estar alojado escuchando el servidor
console.log('El Servidor Está Alojado en el Puerto 3000')