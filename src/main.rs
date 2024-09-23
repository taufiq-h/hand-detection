use opencv::{
    core::{self, Point, Rect, Scalar},
    highgui, imgproc, prelude::*, types, videoio,
};

fn main() -> opencv::Result<()> {
    // Membuka kamera
    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;
    if !videoio::VideoCapture::is_opened(&cam)? {
        panic!("Tidak bisa membuka kamera");
    }

    // Membaca frame dari kamera
    let window = "Tombol Virtual";
    highgui::named_window(window, highgui::WINDOW_AUTOSIZE)?;

    loop {
        let mut frame = core::Mat::default();
        cam.read(&mut frame)?;
        if frame.size()?.width == 0 {
            continue; // Skip jika frame kosong
        }

        // Konversi frame ke grayscale untuk mempermudah deteksi
        let mut gray = core::Mat::default();
        imgproc::cvt_color(&frame, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;

        // Thresholding untuk segmentasi warna kulit
        let mut thresh = core::Mat::default();
        imgproc::threshold(&gray, &mut thresh, 100.0, 255.0, imgproc::THRESH_BINARY_INV)?;  // Sesuaikan threshold 100.0

        // Mendeteksi kontur tangan
        let mut contours = types::VectorOfVectorOfPoint::new();
        let mut hierarchy = core::Mat::default();
        imgproc::find_contours(
            &thresh,
            &mut contours,
            imgproc::RETR_TREE,
            imgproc::CHAIN_APPROX_SIMPLE,
            Point::new(0, 0),
        )?;

        // Gambarkan area tombol virtual di layar
        let tombol_atas_kiri = Point::new(100, 100);
        let tombol_bawah_kanan = Point::new(300, 300);
        let tombol_rect = Rect::new(
            tombol_atas_kiri.x,
            tombol_atas_kiri.y,
            tombol_bawah_kanan.x - tombol_atas_kiri.x,
            tombol_bawah_kanan.y - tombol_atas_kiri.y,
        );
        imgproc::rectangle(
            &mut frame,
            tombol_rect,
            Scalar::new(0.0, 255.0, 0.0, 0.0),
            2,
            imgproc::LINE_8,
            0,
        );

        let mut tombol_ditekan = false;

        // Cek apakah tangan berada di dalam area tombol virtual
        for i in 0..contours.len() {
            let contour = contours.get(i)?;
            let bounding_rect = imgproc::bounding_rect(&contour)?;

            // Filter berdasarkan ukuran bounding box 
            if bounding_rect.width > 50 && bounding_rect.height > 50 {
                // Gambar bounding box untuk kontur yang terdeteksi
                imgproc::rectangle(
                    &mut frame,
                    bounding_rect,
                    Scalar::new(255.0, 0.0, 0.0, 0.0),
                    2,
                    imgproc::LINE_8,
                    0,
                );

                // Cek jika bounding box tangan berada di dalam area tombol virtual
                if bounding_rect.x > tombol_atas_kiri.x
                    && bounding_rect.y > tombol_atas_kiri.y
                    && bounding_rect.x + bounding_rect.width < tombol_bawah_kanan.x
                    && bounding_rect.y + bounding_rect.height < tombol_bawah_kanan.y
                {
                    tombol_ditekan = true;
                }
            }
        }

        if tombol_ditekan {
            println!("Tombol virtual ditekan!");
        } else {
            println!("Tombol virtual tidak ditekan.");
        }

        // Tampilkan frame dengan kontur dan tombol virtual
        highgui::imshow(window, &frame)?;

        // Tekan ESC untuk keluar
        if highgui::wait_key(10)? == 27 {
            break;
        }
    }

    Ok(())
}
