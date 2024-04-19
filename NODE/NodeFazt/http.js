const http = require("http");

// Crear un servidor básico
const server = http.createServer(function (request, response) {
  if (request.url === "/") {
    response.write("BIENVENIDO AL SERVIDOR"); // si la conexion funciona, dar esta respuesta
    return response.end(); // terminar respuesta y ejecucion
  }

  if (request.url === "/about") {
    // si va a la url /about
    response.write("Acerca de"); // si la conexion funciona, dar esta respuesta
    return response.end(); // terminar respuesta y ejecucion
  }

  response.write(`
        <h1>NO FOUND</h1>
        <p>Pagina no encontrada</p>
        <a href="/">ir al Home</a>
    `); // si la conexion funciona, pero no se dirige a una url conocida dar esta respuesta
  response.end(); // terminar respuesta
});

server.listen(3000); // el servidor esta en el puerto #

console.log("SERVIDOR ESCUCHANDO EN EL PUERTO 3000");
