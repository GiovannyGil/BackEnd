import express from 'express'

// importar el router
import diaryRouter from './routes/diaries'

const app = express() // Crear una nueva aplicaicon de express

// usar middleware para parsear el body a json
app.use(express.json()) // para application/json

// puerto
const PORT = 3000

// crear el endpoint()
app.get('/ping', (_req, res) => {
  console.log('alguien hizo un ping aquí')
  res.send('pong')
})

// usar el router
app.use('/api/diaries', diaryRouter)

// levantar el servidor
app.listen(PORT, () => {
  console.log(`Server is running on port: ${PORT}`)
})
