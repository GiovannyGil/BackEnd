import {Schema, model} from "mongoose";
import bycrypt from "bcryptjs"


const UserSchema = new Schema({
    username: {
        type: String,
        unique: true,
    },
    email: {
        type: String,
        unique: true,
    },
    password: {
        type: String,
        required: true,
    },
    roles: [{ // referencia a roles -> una a muchos, un usario puede tener varios roles
        ref: "Role", // relacionado con el modelo de roles
        type: Schema.Types.ObjectId
    }]
},{
    timestamps: true,
    versionKey: false
})

// metodo para cifrar y comparar contraseñas

UserSchema.statics.encryptPassword = async (password) => {
    const salt = await bycrypt.genSalt(10) // aplicar recorrido 10 veces
    // crear contraseña cifrada
    return await bycrypt.hash(password, salt)

}
UserSchema.statics.comparePassword = async (password, receivedPassword) => {
    return await bycrypt.compare(password, receivedPassword) // compara contraseñas y devuelve true o false
}


export default model('User', UserSchema)