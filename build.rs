fn main() {
    let mut res = winres::WindowsResource::new();
    res.set_icon("jiafei.ico");
    res.compile().unwrap();
}
