import { Sequelize, DataTypes, Model } from "sequelize"

// establecer conexión con la base de datos
// la base de datos debe existir
const sequelize = new Sequelize('productTest', 'root', '', {
    host: 'localhost',
    dialect: 'mysql',
    port: 3306
})


// // verificar la conexión
// async function testConnection() {
//     try{
//         await sequelize.authenticate()
//         console.log('Conexión establecida con la base de datos')
//     } catch (error) {
//         console.log('Error al conectar a la base de datos', error)
//     }
// }

// testConnection()

// modelo de la tabla product
class Product extends Model {}
Product.init({
    // definir los campos de la tabla
    ProductID: {
        type: DataTypes.UUID,
        primaryKey: true,
        autoIncrement: true
    },
    name: {
        type: DataTypes.STRING,
        allowNull: false
    },
    price: {
        type: DataTypes.DECIMAL
    },
    description: {
        type: DataTypes.TEXT
    }
}, {
    sequelize,
    modelName: 'Product',
    timestamps: false
})

export default Product