export const queries = {
    getAllProducts: 'SELECT * FROM products',
    addNewProduct: 'INSERT INTO products (Name, Description, Quantity) values (@Name, @Description, @Quantity)',
    getProductById: 'SELECT * FROM products WHERE id = @id',
    deleteProduct: 'DELETE FROM products WHERE id = @id',
    getTotalProducts: 'SELECT COUNT(*) FROM products',
    updateProductById: 'UPDATE products SET Name = @Name, Description = @Description, Quantity = @Quantity WHERE id = @Id'
}