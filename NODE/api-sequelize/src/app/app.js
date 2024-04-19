import express from 'express'
import morgan from 'morgan'
import productRouter from '../router/product.router.js'

const app = express()


app.use(morgan('dev')) // middleware para ver las peticiones que llegan al servidor

app.get('/', (req, res) => {
    res.send('API desde NODE/EXPRESS con Sequelize y MySQL')
})
app.use('/api/', productRouter)

export default app