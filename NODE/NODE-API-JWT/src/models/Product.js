import {Schema, model} from "mongoose";


// definir el esquema de la base de datos -> tabla productos con sus campos
const productSchema =  new Schema({
    name: String,
    category: String,
    price: Number,
    imgURL: String
}, {
    timestamps: true,
    versionKey: false
})

export default model("Product", productSchema)