
// declarar constantes con valores diferentes
const myWebAddress = 'gioDEV.com' // tipo texto
const myNumber = 22 // tipo number
const myArray = [10,20,30] // tipo array
const user = { 
    name : 'Gio',
    lastname : 'Chica'
} // tipo objeto


// declarar un objeto que contenga todos los elementos para posterior exportarlos
/* const group = { // forma numero uno
    myWebAddress : myWebAddress,
    myNumber : myNumber,
    myArray : myArray,
    user : user
} */
const group = {
    myWebAddress, // forma numero dos
    myNumber,
    myArray,
    user
}

module.exports = group


// otra forma de exportar sin declarar una constante
/* module.exports = {
    myWebAddress : myWebAddress,
    myNumber : myNumber,
    myArray : myArray,
    user : user
} */

// exportar individualmente

module.exports.user = user // exporta solo el objeto de user
module.exports.myNumber = myNumber // exporta solo la constante myNumber