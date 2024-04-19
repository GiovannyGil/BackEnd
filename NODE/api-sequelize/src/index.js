import app from './app/app.js'

// definir puerto
const PORT = process.env.PORT || 3000

// escuchar puerto -> iniciar servidor en el puerto 3000
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`)
})