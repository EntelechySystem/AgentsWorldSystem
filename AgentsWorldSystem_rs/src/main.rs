use bevy::prelude::*;

fn main() {
    App::new()
        .add_systems(Update, (hello_world).chain())
        .run();
}

fn hello_world() {
    println!("hello world!");
}
