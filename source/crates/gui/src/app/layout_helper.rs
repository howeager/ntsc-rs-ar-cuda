use eframe::egui::{self, vec2};

pub trait LayoutHelper {
    fn ltr<R>(&mut self, add_contents: impl FnOnce(&mut Self) -> R) -> egui::InnerResponse<R>;
    fn rtl<R>(&mut self, add_contents: impl FnOnce(&mut Self) -> R) -> egui::InnerResponse<R>;
}

fn ui_with_layout<'c, R>(
    ui: &mut egui::Ui,
    layout: egui::Layout,
    add_contents: Box<dyn FnOnce(&mut egui::Ui) -> R + 'c>,
) -> egui::InnerResponse<R> {
    let initial_size = vec2(
        ui.available_size_before_wrap().x,
        ui.spacing().interact_size.y,
    );

    ui.allocate_ui_with_layout(initial_size, layout, |ui| add_contents(ui))
}

impl LayoutHelper for egui::Ui {
    fn ltr<R>(&mut self, add_contents: impl FnOnce(&mut egui::Ui) -> R) -> egui::InnerResponse<R> {
        ui_with_layout(
            self,
            egui::Layout::left_to_right(egui::Align::Center),
            Box::new(add_contents),
        )
    }

    fn rtl<R>(&mut self, add_contents: impl FnOnce(&mut egui::Ui) -> R) -> egui::InnerResponse<R> {
        ui_with_layout(
            self,
            egui::Layout::right_to_left(egui::Align::Center),
            Box::new(add_contents),
        )
    }
}

pub trait PanelExt {
    fn interact_height(self, ui: &egui::Ui) -> Self;
    fn interact_height_tall(self, ui: &egui::Ui) -> Self;
}

// This only makes sense for horizontal panels, but egui shipped their new panel API without providing *any* way to tell
// if a panel is horizontal. Sigh...
impl PanelExt for egui::Panel {
    fn interact_height(self, ui: &egui::Ui) -> Self {
        let mut frame = egui::Frame::side_top_panel(ui.style());
        frame.inner_margin.top = 3;
        frame.inner_margin.bottom = 3;
        self.exact_size(
            ui.style().spacing.interact_size.y
                + frame.inner_margin.sum().y
                + frame.stroke.width * 2.0
                + frame.outer_margin.sum().y,
        )
        .frame(frame)
    }

    fn interact_height_tall(self, ui: &egui::Ui) -> Self {
        let mut frame = egui::Frame::side_top_panel(ui.style());
        frame.inner_margin.top = 0;
        frame.inner_margin.bottom = 0;
        self.exact_size(ui.style().spacing.interact_size.y * 2.0)
            .frame(frame)
    }
}
