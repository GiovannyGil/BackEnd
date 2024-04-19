const http = require('http')


const server = http.createServer((req, res)=>{
    if (req.url === "/") {
        res.write("BIENVENIDO AL SERVIDOR"); // si la conexion funciona, dar esta respuesta
        return res.end(); // terminar respuesta y ejecucion
      }
    
      if (req.url === "/about") { // si va a la url /about
/*         for (let i = 0; i < 1000; i++){
            console.log(Math.random()*i)
        } */
        return res.end("Acerca de"); // terminar respuesta y ejecucion
      }

      res.end('NO FOUND')
})

server.listen(3000) // puerto donde se ejecuta el servidor
console.log('SERVIDOR CORRIENDO EN EL PUERTO 3000')