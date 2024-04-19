// el modulo path permite trabajr con rutas, carpetas y archivos y saber su ubicacion


const path = require('path')

const filePath = path.join('/public', 'dist', '/styles', 'main.css') // path.join -> convierte la ruta a una ruta valida en cualquiero S.O

console.log(path.basename(filePath))
console.log(path.dirname(filePath))
console.log(path.parse(filePath))
console.log(path.resolve('dist')) // resuelve(completa) la ruta
