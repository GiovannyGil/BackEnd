import mongoose from "mongoose";

// Connect to the database
mongoose.connect("mongodb://localhost/companydb", {
    // useNewUrlParser: true,
    // useUnifiedTopology: true,
    // useFindAndModify: true,
    // useCreateIndex: true
})
    .then(() => console.log("Connected to MongoDB"))
    .catch(err => console.error("Could not connect to MongoDB", err));