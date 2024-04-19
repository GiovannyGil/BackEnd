// funcion de sumar
function add(x,y){
    return x+y
}

// funcion de restar
function subtract(x,y){
    return x-y
}

// funcion de multiplicar
export function multiply(x,y){
    return x*y
}

// funcion de dividir
export function divide(x,y){
    return x/y
}


/* console.log(add(10,20))
console.log(subtract(10,20))
console.log(multiply(10,20))
console.log(divide(10,20)) */


// exportar las funciones en un objeto
// module.exports = {
//     add, // exportando la funcion de suma
//     subtract, // exportando la funcion de resta
//     multiply, // exportando la funcion de multiplicacio
//     divide // exportano la funcion de divición
// }
export default { // exportqar con ecmascript -> module en el packa ge.json
    add, // exportando la funcion de suma
    subtract, // exportando la funcion de resta
    multiply, // exportando la funcion de multiplicacio
    divide // exportano la funcion de divición
}
