const web = require('./module/myModule') // extraer o importar -> forma commondJS
const {myArray,user} = require('./module/myModule') // modo para extraer metodos(...) en especifico y no importar todo

//console.log(web) // imprimir los valores del objeto importado


const Maht = require('./Math/index') // importar las funciones de Math

console.log(Maht) // mostrar por consola el objeto y sus valores(funciones) importadas


console.log(Maht.add(10,20)) // mostrar por consola la funcion suma y valores
console.log(Maht.subtract(10,20)) // mostrar por consola la funcion resta y valores
console.log(Maht.multiply(10,20)) // mostrar por consola la funcion multiplicacion y valores
console.log(Maht.divide(10,20)) // mostrar por consola la funcion divicion y valores

