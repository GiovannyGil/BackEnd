import {Schema, model} from "mongoose";

// Definimos el esquema de la colección
export const ROLES = ["user", "admin", "moderator"]

const RoleSchema = new Schema({
    name: String,
}, {
    versionKey: false
})

export default model('Role', RoleSchema)