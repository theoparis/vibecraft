use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let out_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let assets_dir = Path::new(&out_dir).join("assets/textures/block");

    if assets_dir.exists() && fs::read_dir(&assets_dir).unwrap().count() > 10 {
        return;
    }

    if let Err(e) = download_assets(&assets_dir) {
        println!("cargo:warning=Failed to download minecraft assets: {}", e);
        panic!("Failed to download minecraft assets: {}", e);
    }
}

fn download_assets(target_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let client = reqwest::blocking::Client::new();
    let manifest_url = "https://piston-meta.mojang.com/mc/game/version_manifest_v2.json";
    let manifest: serde_json::Value = client.get(manifest_url).send()?.json()?;

    let latest_id = manifest["latest"]["release"]
        .as_str()
        .ok_or("Could not find latest release version")?;

    let versions = manifest["versions"]
        .as_array()
        .ok_or("Invalid versions array")?;

    let version_info = versions
        .iter()
        .find(|v| v["id"].as_str() == Some(latest_id))
        .ok_or("Could not find version info")?;

    let url = version_info["url"].as_str().ok_or("Invalid version url")?;

    let details: serde_json::Value = client.get(url).send()?.json()?;

    let client_jar_url = details["downloads"]["client"]["url"]
        .as_str()
        .ok_or("Could not find client jar url")?;

    let mut jar_response = client.get(client_jar_url).send()?;

    let temp_dir = env::temp_dir();
    let jar_path = temp_dir.join(format!("minecraft_{}.jar", latest_id));
    let mut jar_file = fs::File::create(&jar_path)?;
    io::copy(&mut jar_response, &mut jar_file)?;

    let jar_file = fs::File::open(&jar_path)?;
    let mut zip = zip::ZipArchive::new(jar_file)?;

    fs::create_dir_all(target_dir)?;

    for i in 0..zip.len() {
        let mut file = zip.by_index(i)?;
        let name = file.name().to_string();

        if name.starts_with("assets/minecraft/textures/block/") && name.ends_with(".png") {
            let file_name = Path::new(&name).file_name().unwrap();
            let dest_path = target_dir.join(file_name);
            let mut outfile = fs::File::create(&dest_path)?;
            io::copy(&mut file, &mut outfile)?;
        }
    }

    Ok(())
}
