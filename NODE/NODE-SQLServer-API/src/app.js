import Express from "express"
import config from "./config.js"

import productsRoutes from "./routes/products.routes.js"

// configuracion básica del servidor
const app = Express();

// configurar el puerto 3000 para el servidor
app.set('port', config.port) 

// middleware
app.use(Express.json()) // para que el servidor pueda entender los datos que se envían en formato json
app.use(Express.urlencoded({extended: false})) // para que el servidor pueda entender los datos que se envían desde un formulario

// ruta de products
app.use(productsRoutes)


export default app;