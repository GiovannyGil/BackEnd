using APITransacciones.Context;
using Microsoft.EntityFrameworkCore;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.

// crear variable para la cadena de conexion
var conecxion = builder.Configuration.GetConnectionString("Connection");

// registrar el servicio
builder.Services.AddDbContext<AppDBContext>(options => options.UseSqlServer(conecxion));


builder.Services.AddControllers();
// Learn more about configuring Swagger/OpenAPI at https://aka.ms/aspnetcore/swashbuckle
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();


// enebable cors -> habilitar cors
builder.Services.AddCors(options =>
{
    options.AddPolicy("AllowSpecificOrigin", builder => builder
        .WithOrigins("http://localhost:5173", "http://localhost:4200/") // aqui añadir todos los orgins/urls
        .AllowAnyHeader()
        .AllowAnyMethod()
    );
});

var app = builder.Build();

// Configure the HTTP request pipeline
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

// activar cors
app.UseCors("AllowSpecificOrigin");

app.UseHttpsRedirection();

app.UseAuthorization();

app.MapControllers();

app.Run();
