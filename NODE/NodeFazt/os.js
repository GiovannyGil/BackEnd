const os = require('os')


// el modulo OS permite obtener informacion del sistema opereativo en el que se esta trabajando


// mostrar imformacion por consola en una tabla
console.table({
    os: os.platform(),
    version: os.release(),
    totalMemory: os.totalmem()
})