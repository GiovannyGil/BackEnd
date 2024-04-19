
using BackEnd_API.Dtos;
using BackEnd_API.Repositories;
using Microsoft.EntityFrameworkCore;

namespace BackEnd_API
{
    public class Program
    {
        public static void Main(string[] args)
        {
            var builder = WebApplication.CreateBuilder(args);

            // Add services to the container.

            builder.Services.AddControllers();
            // Learn more about configuring Swagger/OpenAPI at https://aka.ms/aspnetcore/swashbuckle
            builder.Services.AddEndpointsApiExplorer();
            builder.Services.AddSwaggerGen();

            // conexion a la base de datos
            builder.Services.AddDbContext<CustmerDataBaseContext>(builder =>
            {
                // leer la conexion
                builder.UseMySQL("Server=localhost;Post=3306;DataBase=customerdatabase;Uid=root;password=test;");
            });
            var app = builder.Build();

            // Configure the HTTP request pipeline.
            if (app.Environment.IsDevelopment())
            {
                app.UseSwagger();
                app.UseSwaggerUI();
            }

            app.UseHttpsRedirection();

            app.UseAuthorization();


            app.MapControllers();

            app.Run();
        }
    }
}
