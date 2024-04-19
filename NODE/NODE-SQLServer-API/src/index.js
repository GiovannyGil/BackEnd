import app from './app.js'
// iniciar servidor

// configurar el puerto 3000 para el servidor
app.listen(app.get('port')) // se inicia el servidor en el puerto 3000 (o el puerto que se haya configurado en el archivo app.js

console.log('Server on port', app.get('port')); // mensaje en consola para verificar que el servidor está corriendo