import {getConnection, sql, queries} from '../database/index.js';

// metodo para listar todos los productos
export const getProducts = async (req, res) => {
    try{
        // hacer la conexion y oibtener los productos en una promesa
        const pool = await getConnection()
        const result = await pool.request().query(queries.getAllProducts)
        // console.log(result)

        res.json(result)
    }
    catch(error){
        res.status(500)
        res.send(error.message)
    }
}

export const createNewProduct = async (req, res) => {
    const {Name, Description} = req.body // importar los campos del body como constantes
    let {Quantity} = req.body // importar los campos del body como variables(puede reecribirse el valor)

    // validar que los campos no estén vacíos
    if(Name == null || Description == null){
        return res.status(400).json({msg: 'Bad Request. Please fill all fields'})
    }
    // validacion para la cantidad cuando es vacia, darle el valor de cero
    if(Quantity == null) Quantity = 0

    // hacer la consulta

    try{
        const pool = await getConnection()
        await pool.request()
        .input('Name', sql.VarChar, Name)
        .input('Description', sql.Text,Description)
        .input('Quantity', sql.Int, Quantity)
        .query(queries.addNewProduct)

        console.log(Name, Description, Quantity)
        res.json({Name, Description, Quantity})
    }
    catch(error){
        res.status(500)
        res.send(error.message)
    }
}

// metodo para listar un producto por su id
export const getProductById = async (req, res) => {
    const {id} = req.params
    try{
        const pool = await getConnection() // 
        const result = await pool.request()
        .input('id', id)
        .query(queries.getProductById)

        // devolver el resultado, el primero encontrado en posición 0
        res.json(result.recordset[0])
    }
    catch(error){
        res.status(500)
        res.send(error.message)
    }
}

// metodo para eliminar un producto por su id
export const deleteProductById = async (req, res) => {
    const {id} = req.params
    try{
        const pool = await getConnection()
        const result = await pool.request()
        .input('id', id)
        .query(queries.deleteProduct)

        // devolver el resultado, el primero encontrado en posición 0
        res.json({msg: 'Product deleted successfully'})
    }
    catch(error){
        res.status(500)
        res.send(error.message)
    }
}

// metodo para devolver la cantidad de productos(registros) en la tabla
export const getTotalProducts = async (req, res) => {
    try{
        const pool = await getConnection()
        const result = await pool.request().query(queries.getTotalProducts)

        // devolver el resultado, el primero encontrado en posición 0
        res.json(result.recordset[0][''])
        // console.log(result)
    }
    catch(error){
        res.status(500)
        res.send(error.message)
    }
}

// metodo para actualizar un producto por su id
export const updateProductById = async (req, res) => {
    const {Name, Description, Quantity} = req.body
    const {id} = req.params

    if(Name == null || Description == null, Quantity == null){
        return res.status(400).json({msg: 'Bad Request. Please fill all fields'})
    }

    const pool = await getConnection()
    await pool.request()
    .input('Name', sql.VarChar, Name)
    .input('Description', sql.Text, Description)
    .input('Quantity', sql.Int, Quantity)
    .input('id', sql.Int, id)
    .query(queries.updateProductById)

    res.json({Name, Description, Quantity}) // devolver el resultado
}