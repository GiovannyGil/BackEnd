// llamar rutas para obtener un API con FETCH



// fetch('URL...')
//     .then(res=>res.json())
//     .then(data =>console.log(data))


// con async y await

async function loadData(){
    try{
        const res = await fetch('URL...')
        const data = await res.json()
        console.log(data)
    } catch (error){
        console.log(error)
    }
}

loadData()