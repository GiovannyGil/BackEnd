import Express  from "express";
import morgan from "morgan";
import pkg from '../package.json'


// importar las rutas de productos
import productsRoutes from './routes/products.routes'
// importar las rutas de autenticacion
import authRoutes from './routes/auth.routes'
// importar la funcion para crear roles por defecto
import { createRoles } from './libs/initialSetup';
//importar las rutas de los users
import userRoutes from './routes/user.routes'


// iniciar express
const app = Express()
createRoles() // crear roles por defecto


// settings
app.set('pkg', pkg)



// middlewares
app.use(morgan('dev'))

// enable json -> entender los datos que llegan desde el cliente
app.use(Express.json())

app.get('/', (req, res) => {
    res.json({
        author: app.set('pkg').author,
        description: app.set('pkg').description,
        version: app.set('pkg').version
    })
})

app.use('/api/products', productsRoutes)
app.use('/api/auth', authRoutes)
app.use('/api/users', userRoutes)

export default app