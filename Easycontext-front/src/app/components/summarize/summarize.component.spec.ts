import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { SummarizeComponent } from './summarize.component';

describe('SummarizeComponent', () => {
  let component: SummarizeComponent;
  let fixture: ComponentFixture<SummarizeComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ SummarizeComponent ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(SummarizeComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
